/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.controller.impl;

import java.util.List;

import org.edgexfoundry.controller.Action;
import org.edgexfoundry.controller.DeviceController;
import org.edgexfoundry.controller.NotificationClient;
import org.edgexfoundry.dao.AddressableDao;
import org.edgexfoundry.dao.DeviceProfileDao;
import org.edgexfoundry.dao.DeviceReportDao;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.DeviceServiceDao;
import org.edgexfoundry.domain.meta.ActionType;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.AdminState;
import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceProfile;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.domain.meta.OperatingState;
import org.edgexfoundry.exception.controller.ClientException;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.support.domain.notifications.Notification;
import org.edgexfoundry.support.domain.notifications.NotificationCategory;
import org.edgexfoundry.support.domain.notifications.NotificationSeverity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.data.domain.Sort;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/device")
public class DeviceControllerImpl implements DeviceController {

  @Value("${read.max.limit}")
  private int maxLimit;

  private static final org.edgexfoundry.support.logging.client.EdgeXLogger logger =
      org.edgexfoundry.support.logging.client.EdgeXLoggerFactory
          .getEdgeXLogger(DeviceControllerImpl.class);

  private static final String ERR_GET = "Error getting device:  ";

  @Autowired
  private DeviceRepository repos;

  @Autowired
  private AddressableDao addressableDao;

  @Autowired
  private DeviceProfileDao profileDao;

  @Autowired
  private DeviceServiceDao serviceDao;

  @Autowired
  private CallbackExecutor callback;

  @Autowired
  private DeviceReportDao deviceRptDao;

  @Autowired
  private NotificationClient notificationClient;

  @Value("${notification.postdevicechanges}")
  private boolean notifyDeviceChanges;

  @Value("${notification.slug}")
  private String notificationSlug;

  @Value("${notification.content}")
  private String notificationContent;

  @Value("${notification.sender}")
  private String notificationSender;

  @Value("${notification.description}")
  private String notificationDescription;

  @Value("${notification.label}")
  private String notificationLabel;

  /**
   * Fetch a specific device by database generated id. May return null if no device with the id is
   * found. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if device not found by id.
   * 
   * @param String database generated id for the device
   * 
   * @return device matching on the id
   */
  @RequestMapping(value = "/{id}", method = RequestMethod.GET)
  @Override
  public Device device(@PathVariable String id) {
    try {
      Device device = repos.findOne(id);
      if (device == null)
        throw new NotFoundException(Device.class.toString(), id);
      return device;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return all devices sorted by id. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns LimitExceededException (HTTP 413) if the number returned exceeds
   * the max limit.
   * 
   * @return list of device
   */
  @RequestMapping(method = RequestMethod.GET)
  @Override
  public List<Device> devices() {
    try {
      if (repos.count() > maxLimit) {
        logger.error("Max limit exceeded on request for devices");
        throw new LimitExceededException("Device");
      }
      Sort sort = new Sort(Sort.Direction.DESC, "_id");
      return repos.findAll(sort);
    } catch (LimitExceededException lE) {
      throw lE;
    } catch (Exception e) {
      logger.error("Error getting devices:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return Device matching given name (device names should be unique). May be null if no device
   * matches on the name provided. Returns ServiceException (HTTP 503) for unknown or unanticipated
   * issues. Returns NotFoundException (HTTP 404) if device not found by name.
   * 
   * @param name
   * @return device matching on name
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.GET)
  @Override
  public Device deviceForName(@PathVariable String name) {
    try {
      Device device = repos.findByName(name);
      if (device == null)
        throw new NotFoundException(Device.class.toString(), name);
      return device;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find all Devices having at least one label matching the label provided. List may be empty if no
   * device match. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param label - label to be matched
   * 
   * @return List of Device matching on specified label
   */
  @RequestMapping(method = RequestMethod.GET, value = "/label/{label:.+}")
  @Override
  public List<Device> devicesByLabel(@PathVariable String label) {
    try {
      return repos.findByLabelsIn(label);
    } catch (Exception excep) {
      logger.error("Error getting devices:  " + excep.getMessage());
      throw new ServiceException(excep);
    }
  }

  /**
   * Find all devices associated to the DeviceService with the specified DeviceService database
   * generated identifier. List may be empty if no device match. Returns ServiceException (HTTP 503)
   * for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no DeviceService
   * match on the id provided.
   * 
   * @param serviceId - device service's database generated identifier
   * @return List of Devices associated to the device service
   */
  @RequestMapping(method = RequestMethod.GET, value = "/service/{serviceId}")
  @Override
  public List<Device> devicesForService(@PathVariable String serviceId) {
    try {
      DeviceService service = serviceDao.getById(serviceId);
      if (service == null) {
        logger.error("Request for device by non-existent service:  " + serviceId);
        throw new NotFoundException(DeviceService.class.toString(), serviceId);
      }
      return repos.findByService(service);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find all devices associated to the DeviceService with the specified service name (DeviceService
   * names must be unique). List may be empty if no device match. Returns ServiceException (HTTP
   * 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no
   * DeviceService match on the name provided.
   * 
   * @param servicename - device service's name
   * @return List of Devices associated to the device service
   */
  @RequestMapping(method = RequestMethod.GET, value = "/servicename/{servicename:.+}")
  @Override
  public List<Device> devicesForServiceByName(@PathVariable String servicename) {
    try {
      DeviceService service = serviceDao.getByName(servicename);
      if (service == null) {
        logger.error("Request for device by non-existent service name:  " + servicename);
        throw new NotFoundException(DeviceService.class.toString(), servicename);
      }
      return repos.findByService(service);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find all devices associated to the DeviceProfile with the specified profile database generated
   * identifier. List may be empty if no device match. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no DeviceProfile match
   * on the id provided.
   * 
   * @param profile id - device profile's database generated identifier
   * @return List of Devices associated to the device profile
   */
  @RequestMapping(method = RequestMethod.GET, value = "/profile/{profileId}")
  @Override
  public List<Device> devicesForProfile(@PathVariable String profileId) {
    try {
      DeviceProfile profile = profileDao.getById(profileId);
      if (profile == null) {
        logger.error("Request for device by non-existent profile:  " + profileId);
        throw new NotFoundException(DeviceProfile.class.toString(), profileId);
      }
      return repos.findByProfile(profile);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception excep) {
      logger.error(ERR_GET + excep.getMessage());
      throw new ServiceException(excep);
    }
  }

  /**
   * Find all devices associated to the DeviceProfile with the specified profile name. List may be
   * empty if no device match. Returns ServiceException (HTTP 503) for unknown or unanticipated
   * issues. Returns NotFoundException (HTTP 404) if no DeviceProfile match on the name provided.
   * 
   * @param profile name - device profile's name
   * @return List of Devices associated to the device profile
   */
  @RequestMapping(method = RequestMethod.GET, value = "/profilename/{profilename:.+}")
  @Override
  public List<Device> devicesForProfileByName(@PathVariable String profilename) {
    try {
      DeviceProfile profile = profileDao.getByName(profilename);
      if (profile == null) {
        logger.error("Request for device by non-existent profile:  " + profilename);
        throw new NotFoundException(DeviceProfile.class.toString(), profilename);
      }
      return repos.findByProfile(profile);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception excep) {
      logger.error(ERR_GET + excep.getMessage());
      throw new ServiceException(excep);
    }
  }

  /**
   * Find all devices associated to the Addressable with the specified addressable database
   * generated identifier. List may be empty if no device match. Returns ServiceException (HTTP 503)
   * for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no Addressable
   * match on the id provided.
   * 
   * @param addressable id - addressable's database generated identifier
   * @return List of Devices associated to the addressable
   */
  @RequestMapping(method = RequestMethod.GET, value = "/addressable/{addressableId}")
  @Override
  public List<Device> devicesForAddressable(@PathVariable String addressableId) {
    try {
      Addressable addressable = addressableDao.getById(addressableId);
      if (addressable == null) {
        logger.error("Request for device by non-existent addressable:  " + addressableId);
        throw new NotFoundException(Addressable.class.toString(), addressableId);
      }
      return repos.findByAddressable(addressable);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find all devices associated to the Addressable with the specified addressable name. List may be
   * empty if no device match. Returns ServiceException (HTTP 503) for unknown or unanticipated
   * issues. Returns NotFoundException (HTTP 404) if no Addressable match on the name provided.
   * 
   * @param addressable name - addressable's name
   * @return List of Devices associated to the addressable
   */
  @RequestMapping(method = RequestMethod.GET, value = "/addressablename/{addressablename:.+}")
  @Override
  public List<Device> devicesForAddressableByName(@PathVariable String addressablename) {
    try {
      Addressable addressable = addressableDao.getByName(addressablename);
      if (addressable == null) {
        logger.error("Request for device by non-existent addressable:  " + addressablename);
        throw new NotFoundException(Addressable.class.toString(), addressablename);
      }
      return repos.findByAddressable(addressable);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Add a new Device - name must be unique. Embedded objects (device, service, profile,
   * addressable) are all referenced in the new Device object by id or name to associated objects.
   * All other data in the embedded objects will be ignored. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues. Returns DataValidationException (HTTP 409) if an associated
   * object (Addressable, Profile, Service) cannot be found with the id or name provided.
   * 
   * @param Device object
   * @return database generated identifier for the new device
   */
  @RequestMapping(method = RequestMethod.POST)
  @Override
  public String add(@RequestBody Device device) {
    if (device == null)
      throw new ServiceException(new DataValidationException("No device data provided"));
    if (device.getAdminState() == null || device.getOperatingState() == null)
      throw new DataValidationException("Device and Admin state cannot be null");
    try {
      attachAssociated(device);
      repos.save(device);
      notifyAssociates(device, Action.POST);
      return device.getId();
    } catch (DuplicateKeyException dE) {
      throw new DataValidationException("Name is not unique: " + device.getName());
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error adding device:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private void attachAssociated(Device device) {
    DeviceService service = serviceDao.getByIdOrName(device.getService());
    if (service == null)
      throw new DataValidationException("A device must be associated to a known device service.");
    device.setService(service);
    DeviceProfile profile = profileDao.getByIdOrName(device.getProfile());
    if (profile == null)
      throw new DataValidationException("A device must be associated to a known device profile.");
    device.setProfile(profile);
    Addressable addressable = addressableDao.getByIdOrName(device.getAddressable());
    if (addressable == null)
      throw new DataValidationException("A device must be associated to a known addressable.");
    device.setAddressable(addressable);
  }

  /**
   * Update the last connected time of the device by database generated identifier. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device cannot be found by the identifier provided.
   * 
   * @param id - database generated identifier for the device
   * @param time - new last connected time in milliseconds
   * @return boolean indicating success of update
   */
  @RequestMapping(value = "/{id}/lastconnected/{time}", method = RequestMethod.PUT)
  @Override
  public boolean updateLastConnected(@PathVariable String id, @PathVariable long time) {
    return updateLastConnected(id, time, false);
  }

  /**
   * Update the last connected time of the device by database generated identifier. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device cannot be found by the identifier provided.
   * 
   * @param id - database generated identifier for the device
   * @param time - new last connected time in milliseconds
   * @param notify - boolean indicating whether callback should be made to device service to inform
   *        of update
   * @return boolean indicating success of update
   */
  @RequestMapping(value = "/{id}/lastconnected/{time}/{notify}", method = RequestMethod.PUT)
  @Override
  public boolean updateLastConnected(@PathVariable String id, @PathVariable long time,
      @PathVariable boolean notify) {
    try {
      Device device = repos.findOne(id);
      if (device == null) {
        logger.error("Request to update last connected time with non-existent device:  " + id);
        throw new NotFoundException(Device.class.toString(), id);
      }
      return updateLastConnected(device, time, notify);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  // added notify param per TDC/Cloud suggestion since most of the time the
  // device service is the one reporting and doesn't need the last connected
  // to be sent back on callback.
  private boolean updateLastConnected(Device device, long time, boolean notify) {
    try {
      device.setLastConnected(time);
      repos.save(device);
      if (notify)
        notifyAssociates(device, Action.PUT);
      return true;
    } catch (Exception e) {
      logger.error("Error updating last connected time for the device:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the last connected time of the device by unique name of the device. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device cannot be found by the name provided.
   * 
   * @param name - device name
   * @param time - new last connected time in milliseconds
   * @return boolean indicating success of update
   */
  @RequestMapping(value = "/name/{name:.+}/lastconnected/{time}", method = RequestMethod.PUT)
  @Override
  public boolean updateLastConnectedByName(@PathVariable String name, @PathVariable long time) {
    return updateLastConnectedByName(name, time, false);
  }

  /**
   * Update the last connected time of the device by unique name of the device. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device cannot be found by the name provided.
   * 
   * @param name - device name
   * @param time - new last connected time in milliseconds
   * @param notify - boolean indicating whether callback should be made to device service to inform
   *        of update
   * 
   * @return boolean indicating success of update
   */
  @RequestMapping(value = "/name/{name:.+}/lastconnected/{time}/{notify}",
      method = RequestMethod.PUT)
  @Override
  public boolean updateLastConnectedByName(@PathVariable String name, @PathVariable long time,
      @PathVariable boolean notify) {
    try {
      Device device = repos.findByName(name);
      if (device == null) {
        logger.error("Request to update last connected time with non-existent device:  " + name);
        throw new NotFoundException(Device.class.toString(), name);
      }
      return updateLastConnected(device, time, notify);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the last reported time of the device by database generated identifier. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device cannot be found by the identifier provided.
   * 
   * @param id - database generated identifier for the device
   * @param time - new last reported time in milliseconds
   * @return boolean indicating success of update
   */
  @RequestMapping(value = "/{id}/lastreported/{time}", method = RequestMethod.PUT)
  @Override
  public boolean updateLastReported(@PathVariable String id, @PathVariable long time) {
    return updateLastReported(id, time, false);
  }

  /**
   * Update the last reported time of the device by database generated identifier. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device cannot be found by the identifier provided.
   * 
   * @param id - database generated identifier for the device
   * @param time - new last reported time in milliseconds
   * @param notify - boolean indicating whether callback should be made to device service to inform
   *        of update
   * 
   * @return boolean indicating success of update
   */
  @RequestMapping(value = "/{id}/lastreported/{time}/{notify}", method = RequestMethod.PUT)
  @Override
  public boolean updateLastReported(@PathVariable String id, @PathVariable long time,
      @PathVariable boolean notify) {
    try {
      Device device = repos.findOne(id);
      if (device == null) {
        logger.error("Request to update last reported time with non-existent device:  " + id);
        throw new NotFoundException(Device.class.toString(), id);
      }
      return updateLastReported(device, time, notify);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  // added notify param per TDC/Cloud suggestion since most of the time the
  // device service is the one reporting and doesn't need the last connected
  // to be sent back on callback.
  private boolean updateLastReported(Device device, long time, boolean notify) {
    try {
      device.setLastReported(time);
      repos.save(device);
      if (notify)
        notifyAssociates(device, Action.PUT);
      return true;
    } catch (Exception e) {
      logger.error("Error updating last reported time for the device:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the last reported time of the device by unique name of the device. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device cannot be found by the name provided.
   * 
   * @param name - device name
   * @param time - new last reported time in milliseconds
   * @return boolean indicating success of update
   */
  @RequestMapping(value = "/name/{name:.+}/lastreported/{time}", method = RequestMethod.PUT)
  @Override
  public boolean updateLastReportedByName(@PathVariable String name, @PathVariable long time) {
    return updateLastReportedByName(name, time, false);
  }

  /**
   * Update the last reported time of the device by unique name of the device. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device cannot be found by the name provided.
   * 
   * @param name - device name
   * @param time - new last reported time in milliseconds
   * @param notify - boolean indicating whether callback should be made to device service to inform
   *        of update
   * 
   * @return boolean indicating success of update
   */
  @RequestMapping(value = "/name/{name:.+}/lastreported/{time}/{notify}",
      method = RequestMethod.PUT)
  @Override
  public boolean updateLastReportedByName(@PathVariable String name, @PathVariable long time,
      @PathVariable boolean notify) {
    try {
      Device device = repos.findByName(name);
      if (device == null) {
        logger.error("Request to update last reported time with non-existent device:  " + name);
        throw new NotFoundException(Device.class.toString(), name);
      }
      return updateLastReported(device, time, notify);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the op state of the device by database generated identifier. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if the
   * device cannot be found by the identifier provided. Returns DataValidationException (HTTP 409)
   * if the proposed new state is null;
   * 
   * @param id - database generated identifier for the device
   * @param opState - new op state for the device (either ENABLED or DISABLED)
   * @return - boolean indicating success of the operation
   */
  @RequestMapping(value = "/{id}/opstate/{opState}", method = RequestMethod.PUT)
  @Override
  public boolean updateOpState(@PathVariable String id, @PathVariable String opState) {
    try {
      Device device = repos.findOne(id);
      if (device == null) {
        logger.error("Request to update op state with non-existent device:  " + id);
        throw new NotFoundException(Device.class.toString(), id);
      }
      return updateOpState(device, opState);
    } catch (DataValidationException dE) {
      throw dE;
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private boolean updateOpState(Device device, String state) {
    if (state == null)
      throw new DataValidationException("Op state cannot be set to null");
    try {
      device.setOperatingState(OperatingState.valueOf(state));
      repos.save(device);
      notifyAssociates(device, Action.PUT);
      return true;
    } catch (Exception e) {
      logger.error("Error updating op state for the device:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the op status time of the device by unique name of the device. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if the
   * device cannot be found by the name provided. Returns DataValidationException (HTTP 409) if the
   * proposed new state is null;
   * 
   * @param name - device name
   * @param opState - new op state for the device (either ENABLED or DISABLED)
   * @return - boolean indicating success of the operation
   */
  @RequestMapping(value = "/name/{name:.+}/opstate/{opState}", method = RequestMethod.PUT)
  @Override
  public boolean updateOpStateByName(@PathVariable String name, @PathVariable String opState) {
    try {
      Device device = repos.findByName(name);
      if (device == null) {
        logger.error("Request to update op state with non-existent device name:  " + name);
        throw new NotFoundException(Device.class.toString(), name);
      }
      return updateOpState(device, opState);
    } catch (DataValidationException dE) {
      throw dE;
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the admin state of the device by database generated identifier. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if the
   * device cannot be found by the identifier provided. Returns DataValidationException (HTTP 409)
   * if the proposed new state is null;
   * 
   * @param id - database generated identifier for the device
   * @param adminstate - new admin state for the device (either LOCKED or UNLOCKED)
   * @return - boolean indicating success of the operation
   */
  @RequestMapping(value = "/{id}/adminstate/{adminState}", method = RequestMethod.PUT)
  @Override
  public boolean updateAdminState(@PathVariable String id, @PathVariable String adminState) {
    try {
      Device device = repos.findOne(id);
      if (device == null) {
        logger.error("Request to update admin state with non-existent device:  " + id);
        throw new NotFoundException(Device.class.toString(), id);
      }
      boolean success = updateAdminState(device, adminState);
      notifyAssociates(device, Action.PUT);
      return success;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private boolean updateAdminState(Device device, String state) {
    if (state == null)
      throw new DataValidationException("Admin state cannot be set to null");
    try {
      device.setAdminState(AdminState.valueOf(state));
      repos.save(device);
      return true;
    } catch (Exception e) {
      logger.error("Error updating admin state for the device:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the admin state of the device by device name. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if the device cannot be
   * found by the identifier provided. Returns DataValidationException (HTTP 409) if the proposed
   * new state is null;
   * 
   * @param name - device name
   * @param opState - new admin state for the device (either LOCKED or UNLOCKED)
   * @return - boolean indicating success of the operation
   */
  @RequestMapping(value = "/name/{name:.+}/adminstate/{adminState}", method = RequestMethod.PUT)
  @Override
  public boolean updateAdminStateByName(@PathVariable String name,
      @PathVariable String adminState) {
    try {
      Device device = repos.findByName(name);
      if (device == null) {
        logger.error("Request to update admin state with non-existent device name:  " + name);
        throw new NotFoundException(Device.class.toString(), name);
      }
      boolean success = updateAdminState(device, adminState);
      notifyAssociates(device, Action.PUT);
      return success;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the Device identified by the id or name stored in the object provided. Id is used first,
   * name is used second for identification purposes. New device services & profiles cannot be
   * created with a PUT, but the service and profile can replaced by referring to a new device
   * service or profile id or name. Returns ServiceException (HTTP 503) for unknown or unanticipated
   * issues. Returns NotFoundException (HTTP 404) if the device cannot be found by the identifier
   * provided
   * 
   * @param device2 - object holding the identifier and new values for the Device
   * @return boolean indicating success of the update
   */
  @RequestMapping(method = RequestMethod.PUT)
  @Override
  public boolean update(@RequestBody Device device2) {
    if (device2 == null)
      throw new ServiceException(new DataValidationException("No device data provided"));
    try {
      Device device = getDeviceByIdOrName(device2);
      if (device == null) {
        logger.error("Request to update with non-existent or unidentified device (id/name):  "
            + device2.getId() + "/" + device2.getName());
        throw new NotFoundException(Device.class.toString(), device2.getId());
      }
      updateDevice(device2, device);
      notifyAssociates(device, Action.PUT);
      return true;
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (ClientException cE) {
      throw cE;
    } catch (Exception e) {
      logger.error("Error updating device:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private Device getDeviceByIdOrName(Device device) {
    if (device.getId() != null)
      return repos.findOne(device.getId());
    return repos.findByName(device.getName());
  }

  private void updateDevice(Device from, Device to) {
    if (from.getAddressable() != null)
      to.setAddressable(addressableDao.getByIdOrName(from.getAddressable()));
    if (from.getService() != null)
      to.setService(serviceDao.getByIdOrName(from.getService()));
    if (from.getProfile() != null)
      to.setProfile(profileDao.getByIdOrName(from.getProfile()));
    if (from.getAdminState() != null)
      to.setAdminState(from.getAdminState());
    if (from.getDescription() != null)
      to.setDescription(from.getDescription());
    if (from.getLabels() != null)
      to.setLabels(from.getLabels());
    if (from.getLastConnected() != 0)
      to.setLastConnected(from.getLastConnected());
    if (from.getLastReported() != 0)
      to.setLastReported(from.getLastReported());
    if (from.getLocation() != null)
      to.setLocation(from.getLocation());
    if (from.getOperatingState() != null)
      to.setOperatingState(from.getOperatingState());
    if (from.getOrigin() != 0)
      to.setOrigin(from.getOrigin());
    if (from.getName() != null)
      to.setName(from.getName());
    repos.save(to);
  }

  /**
   * Remove the Device designated by database generated id. This does not remove associated objects
   * (addressable, service, profile, etc.). Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) if the device cannot be found by the
   * identifier provided.
   * 
   * @param database generated id for the device
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/id/{id}", method = RequestMethod.DELETE)
  @Override
  public boolean delete(@PathVariable String id) {
    try {
      Device device = repos.findOne(id);
      if (device == null) {
        logger.error("Request to delete with non-existent device by id:  " + id);
        throw new NotFoundException(Device.class.toString(), id);
      }
      return deleteDevice(device);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error removing device:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Remove the Device designated by unique name. This does not remove associated objects
   * (addressable, service, profile, etc.). Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) if the device cannot be found by the
   * identifier provided.
   * 
   * @param unique name of the device
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.DELETE)
  @Override
  public boolean deleteByName(@PathVariable String name) {
    try {
      Device device = repos.findByName(name);
      if (device == null) {
        logger.error("Request to delete with unknown device by name:  " + name);
        throw new NotFoundException(Device.class.toString(), name);
      }
      return deleteDevice(device);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error removing device:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private boolean deleteDevice(Device device) {
    notifyAssociates(device, Action.DELETE);
    deviceRptDao.removeAssociatedReportsForDevice(device);
    repos.delete(device);
    return true;
  }

  private void notifyAssociates(Device device, Action action) {
    postNotification(device.getName(), action);
    callback.callback(device.getService(), device.getId(), action, ActionType.DEVICE);
  }

  // TODO possibly do async someday
  private void postNotification(String name, Action action) {
    if (notifyDeviceChanges) {
      Notification notification = new Notification();
      notification.setSlug(notificationSlug + System.currentTimeMillis());
      notification.setContent(notificationContent + name + "-" + action);
      notification.setCategory(NotificationCategory.SW_HEALTH);
      notification.setDescription(notificationDescription);
      String[] labels = new String[1];
      labels[0] = notificationLabel;
      notification.setLabels(labels);
      notification.setSender(notificationSender);
      notification.setSeverity(NotificationSeverity.NORMAL);
      postNotification(notification);
    }
  }

  private void postNotification(Notification notification) {
    try {
      notificationClient.receiveNotification(notification);
      logger.debug("Notification sent about new device provisioning:" + notification.getSlug());
    } catch (Exception e) {
      // don't want the send of this notification to disrupt device
      // metadata persistence activity so no retrhow
      logger.debug("Notification send on device modification failed for:" + notification.getSlug()
          + "; " + e);
    }
  }
}
