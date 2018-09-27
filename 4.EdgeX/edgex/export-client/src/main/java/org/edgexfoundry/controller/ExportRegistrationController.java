/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @microservice:  export-client
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/
package org.edgexfoundry.controller;

import java.util.List;
import java.util.stream.Stream;

import org.edgexfoundry.dao.ExportRegistrationRepository;
import org.edgexfoundry.domain.export.ExportCompression;
import org.edgexfoundry.domain.export.ExportDestination;
import org.edgexfoundry.domain.export.ExportEncryption;
import org.edgexfoundry.domain.export.ExportFormat;
import org.edgexfoundry.domain.export.ExportRegistration;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.support.domain.notifications.Notification;
import org.edgexfoundry.support.domain.notifications.NotificationCategory;
import org.edgexfoundry.support.domain.notifications.NotificationSeverity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@CrossOrigin(origins = "*")
@RestController
@RequestMapping("/api/v1/registration")
public class ExportRegistrationController {

	// private static final Logger logger =
	// Logger.getLogger(ExportRegistrationController.class);
	// replace above logger with EdgeXLogger below
	private final static org.edgexfoundry.support.logging.client.EdgeXLogger logger = 
			org.edgexfoundry.support.logging.client.EdgeXLoggerFactory.getEdgeXLogger(ExportRegistrationController.class);

	@Value("${notification.postclientchanges}")
	private boolean notifyClientChanges;

	@Value("${notification.newclient.slug}")
	private String clientAddSlug;

	@Value("${notification.deleteclient.slug}")
	private String clientRemoveSlug;

	@Value("${notification.newclient.content}")
	private String clientAddContent;

	@Value("${notification.deleteclient.content}")
	private String clientRemoveContent;

	@Value("${notification.sender}")
	private String notificationSender;

	@Value("${notification.description}")
	private String notificationDescription;

	@Value("${notification.label}")
	private String notificationLabel;

	@Autowired
	private ExportRegistrationRepository repos;

	@Autowired
	private NotificationClient notiClient;

	/**
	 * Fetch a client export registration by id. Response will be null if no
	 * export registration matches on id. Return ServcieException (HTTP 503) for
	 * unknown or unanticipated issues.
	 * 
	 * @param id
	 *            - the database generated identifier used to request the export
	 *            registration
	 * @return the export registration matching the identifier provided
	 */
	@RequestMapping(value = "/{id}", method = RequestMethod.GET)
	public ExportRegistration exportRegistration(@PathVariable String id) {
		try {
			return repos.findOne(id);
		} catch (Exception e) {
			logger.error("Error getting export registration:  " + e.getMessage());
			throw new ServiceException(e);
		}
	}

	/**
	 * Provide the Export reference list as a string array (based on type
	 * provided) to registration client tools
	 * 
	 * @param type
	 *            - string either algorithms, compressions, formats or
	 *            destinations
	 * @return list of allowable export enum types
	 */
	@RequestMapping(value = "/reference/{type}")
	public String[] referenceOptions(@PathVariable String type) {
		switch (type) {
		case "algorithms":
			return Stream.of(ExportEncryption.values()).map(ExportEncryption::name).toArray(String[]::new);
		case "compressions":
			return Stream.of(ExportCompression.values()).map(ExportCompression::name).toArray(String[]::new);
		case "formats":
			return Stream.of(ExportFormat.values()).map(ExportFormat::name).toArray(String[]::new);
		case "destinations":
			return Stream.of(ExportDestination.values()).map(ExportDestination::name).toArray(String[]::new);
		default:
			logger.error("Error getting reference data for unknown type:  " + type);
			throw new ServiceException(new Exception("Unknown enum type"));
		}
	}

	/**
	 * Fetch all client export registrations. Return ServcieException (HTTP 503)
	 * for unknown or unanticipated issues. No limits are exercised on this
	 * query at this time. May need to add this in the future if the number of
	 * clients is huge.
	 * 
	 * @return - a list of all client export registrations
	 */
	@RequestMapping(method = RequestMethod.GET)
	public List<ExportRegistration> exportRegistrations() {
		try {
			return repos.findAll();
		} catch (Exception e) {
			logger.error("Error getting export registrationsr:  " + e.getMessage());
			throw new ServiceException(e);
		}
	}

	/**
	 * Fetch a client export registration by unique name. Response will be null
	 * if no export registration matches on name. Return ServcieException (HTTP
	 * 503) for unknown or unanticipated issues.
	 * 
	 * @param id
	 *            - the database generated identifier used to request the export
	 *            registration
	 * @return the export registration matching the identifier provided
	 */
	@RequestMapping(value = "/name/{name:.+}", method = RequestMethod.GET)
	public ExportRegistration exportRegistrationByName(@PathVariable String name) {
		try {
			return repos.findByName(name);
		} catch (Exception e) {
			logger.error("Error getting export registration:  " + e.getMessage());
			throw new ServiceException(e);
		}
	}

	/**
	 * Add a new client export registration. Name must be unique across the
	 * database. Return ServcieException (HTTP 503) for unknown or unanticipated
	 * issues.
	 * 
	 * @param exportRegistration
	 * @return - the database generated id for the new export registration.
	 */
	@RequestMapping(method = RequestMethod.POST)
	public String add(@RequestBody ExportRegistration exportRegistration) {
		try {
			checkAndSetDefaults(exportRegistration);
			repos.save(exportRegistration);
			postClientNotification(exportRegistration.getName(), clientAddSlug, clientAddContent);
			return exportRegistration.getId();
		} catch (Exception e) {
			logger.error("Error adding export registration:  " + e.getMessage());
			throw new ServiceException(e);
		}
	}

	private void checkAndSetDefaults(ExportRegistration exportRegistration) {
		if (exportRegistration.getCompression() == null)
			exportRegistration.setCompression(ExportCompression.NONE);
		if (exportRegistration.getFormat() == null)
			exportRegistration.setFormat(ExportFormat.JSON);
		if (exportRegistration.getDestination() == null)
			exportRegistration.setDestination(ExportDestination.MQTT_TOPIC);
		if (exportRegistration.getEncryption() != null
				&& exportRegistration.getEncryption().getEncryptionAlgorithm() == null)
			exportRegistration.getEncryption().setEncryptionAlgorithm(ExportEncryption.NONE);
	}

	/**
	 * Update a client export registration. Name & id are not updated as they
	 * are identifiers. Return NotFoundException (HTTP 404) if the existing
	 * export registration cannot be found by id or name. Return
	 * ServcieException (HTTP 503) for unknown or unanticipated issues.
	 * 
	 * @param exportRegistration
	 * @return - boolean indicating success of the operation.
	 */
	@RequestMapping(method = RequestMethod.PUT)
	public boolean update(@RequestBody ExportRegistration exportRegistration2) {
		try {
			ExportRegistration exportRegistration = getExportRegistrationByIdOrName(exportRegistration2.getId(),
					exportRegistration2.getName());
			if (exportRegistration != null) {
				updateExportRegistration(exportRegistration2, exportRegistration);
				return true;
			} else {
				logger.error("Request to update with non-existent or unidentified export registration (id/name):  "
						+ exportRegistration2.getId() + "/" + exportRegistration2.getName());
				throw new NotFoundException(ExportRegistration.class.toString(), exportRegistration2.getId());
			}
		} catch (NotFoundException nE) {
			throw nE;
		} catch (Exception e) {
			logger.error("Error updating export registration:  " + e.getMessage());
			throw new ServiceException(e);
		}
	}

	/**
	 * Delete a client export registration by database id. Return
	 * NotFoundException (HTTP 404) if the existing export registration cannot
	 * be found by id. Return ServcieException (HTTP 503) for unknown or
	 * unanticipated issues.
	 * 
	 * @param id
	 *            - database generated id for the ExportRegistration
	 * @return - boolean indicating success of the operation
	 */
	@RequestMapping(value = "/id/{id}", method = RequestMethod.DELETE)
	public boolean delete(@PathVariable String id) {
		try {
			ExportRegistration exportRegistration = repos.findOne(id);
			if (exportRegistration != null) {
				repos.delete(exportRegistration);
				postClientNotification(exportRegistration.getName(), clientRemoveSlug, clientRemoveContent);
				return true;
			} else {
				logger.error("Request to delete with non-existent export registration id:  " + id);
				throw new NotFoundException(ExportRegistration.class.toString(), id);
			}
		} catch (NotFoundException nE) {
			throw nE;
		} catch (Exception e) {
			logger.error("Error removing export registration:  " + e.getMessage());
			throw new ServiceException(e);
		}
	}

	/**
	 * Delete a client export registration by unique name. Return
	 * NotFoundException (HTTP 404) if the existing export registration cannot
	 * be found by name. Return ServcieException (HTTP 503) for unknown or
	 * unanticipated issues.
	 * 
	 * @param name
	 *            - unique name for the ExportRegistration
	 * @return - boolean indicating success of the operation
	 */
	@RequestMapping(value = "/name/{name:.+}", method = RequestMethod.DELETE)
	public boolean deleteByName(@PathVariable String name) {
		try {
			ExportRegistration exportRegistration = repos.findByName(name);
			if (exportRegistration != null) {
				repos.delete(exportRegistration);
				postClientNotification(exportRegistration.getName(), clientRemoveSlug, clientRemoveContent);
				return true;
			} else {
				logger.error("Request to delete with unknown export registration name:  " + name);
				throw new NotFoundException(ExportRegistration.class.toString(), name);
			}
		} catch (NotFoundException nE) {
			throw nE;
		} catch (Exception e) {
			logger.error("Error removing export registration:  " + e.getMessage());
			throw new ServiceException(e);
		}
	}

	private void updateExportRegistration(ExportRegistration from, ExportRegistration to) {
		to.setEnable(from.isEnable());
		if (from.getAddressable() != null)
			to.setAddressable(from.getAddressable());
		if (from.getCompression() != null)
			to.setCompression(from.getCompression());
		if (from.getEncryption() != null)
			to.setEncryption(from.getEncryption());
		if (from.getFilter() != null)
			to.setFilter(from.getFilter());
		if (from.getFormat() != null)
			to.setFormat(from.getFormat());
		if (from.getOrigin() != 0)
			to.setOrigin(from.getOrigin());
		if (from.getDestination() != null)
			to.setDestination(from.getDestination());
		repos.save(to);
	}

	private ExportRegistration getExportRegistrationByIdOrName(String id, String name) {
		if (id != null)
			return repos.findOne(id);
		return repos.findByName(name);
	}

	private void postClientNotification(String name, String slug, String content) {
		if (notifyClientChanges) {
			Notification n = new Notification();
			n.setSlug(slug + System.currentTimeMillis());
			n.setContent(content + name);
			n.setCategory(NotificationCategory.SW_HEALTH);
			n.setDescription(notificationDescription);
			String[] labels = new String[1];
			labels[0] = notificationLabel;
			n.setLabels(labels);
			n.setSender(notificationSender);
			n.setSeverity(NotificationSeverity.NORMAL);
			postNotification(n);
		}
	}

	private void postNotification(Notification notification) {
		notiClient.receiveNotification(notification);
		logger.debug("Notification sent about client registration:" + notification.getSlug());
	}

}
