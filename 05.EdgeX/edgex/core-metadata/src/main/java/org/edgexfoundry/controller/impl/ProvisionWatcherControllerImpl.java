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
import org.edgexfoundry.controller.ProvisionWatcherController;
import org.edgexfoundry.dao.DeviceProfileDao;
import org.edgexfoundry.dao.DeviceServiceDao;
import org.edgexfoundry.dao.ProvisionWatcherRepository;
import org.edgexfoundry.domain.meta.ActionType;
import org.edgexfoundry.domain.meta.DeviceProfile;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.domain.meta.ProvisionWatcher;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
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
@RequestMapping("/api/v1/provisionwatcher")
public class ProvisionWatcherControllerImpl implements ProvisionWatcherController {

  private static final org.edgexfoundry.support.logging.client.EdgeXLogger logger =
      org.edgexfoundry.support.logging.client.EdgeXLoggerFactory
          .getEdgeXLogger(ProvisionWatcherControllerImpl.class);

  private static final String ERR_GET = "Error getting provision watcher:  ";

  @Autowired
  ProvisionWatcherRepository repos;

  @Autowired
  private DeviceProfileDao profileDao;

  @Autowired
  private DeviceServiceDao serviceDao;

  @Autowired
  private CallbackExecutor callback;

  @Value("${read.max.limit}")
  private int maxLimit;

  /**
   * Fetch a specific Provision Watcher by database generated id. May return null if no
   * ProvisionWatcher matches on id. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Return NotFoundException (HTTP 404) if no ProvisionWatcher is found by
   * the id provided;
   * 
   * @param String ProvisionWatcher database generated id
   * @return ProvisionWatcher
   */
  @RequestMapping(value = "/{id}", method = RequestMethod.GET)
  @Override
  public ProvisionWatcher watcher(@PathVariable String id) {
    try {
      ProvisionWatcher watcher = repos.findOne(id);
      if (watcher == null)
        throw new NotFoundException(ProvisionWatcher.class.toString(), id);
      return watcher;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return all provision watcher objects sorted by database generated id. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues. Returns LimitExceededException (HTTP 413) if
   * the number returned exceeds the max limit.
   * 
   * @return list of provision watcher
   */
  @RequestMapping(method = RequestMethod.GET)
  @Override
  public List<ProvisionWatcher> watchers() {
    try {
      if (repos.count() > maxLimit) {
        logger.error("Max limit exceeded requesting provision watchers");
        throw new LimitExceededException("ProvisionWatcher");
      } else {
        Sort sort = new Sort(Sort.Direction.DESC, "_id");
        return repos.findAll(sort);
      }
    } catch (LimitExceededException lE) {
      throw lE;
    } catch (Exception e) {
      logger.error("Error getting provision watchers:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return ProvisionWatcher with matching name (name should be unique). May be null if none match.
   * Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Return
   * NotFoundException (HTTP 404) if no ProvisionWatcher is found by the name provided;
   * 
   * @param name
   * @return provision watcher matching on name
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.GET)
  @Override
  public ProvisionWatcher watcherForName(@PathVariable String name) {
    try {
      ProvisionWatcher watcher = repos.findByName(name);
      if (watcher == null)
        throw new NotFoundException(ProvisionWatcher.class.toString(), name);
      return watcher;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find all provision watchers associated to the DeviceProfile with the specified profile database
   * generated identifier. List may be empty if no provision watchers match. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if no DeviceProfile match on the id provided.
   * 
   * @param profile id - device profile's database generated identifier
   * @return List of ProvisionWatchers associated to the device profile
   */
  @RequestMapping(method = RequestMethod.GET, value = "/profile/{profileId}")
  @Override
  public List<ProvisionWatcher> watchersForProfile(@PathVariable String profileId) {
    try {
      DeviceProfile profile = profileDao.getById(profileId);
      if (profile == null) {
        logger.error("Request for provision watcher by non-existent profile:  " + profileId);
        throw new NotFoundException(DeviceProfile.class.toString(), profileId);
      }
      return repos.findByProfile(profile);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error("Error getting provision watchers by profile:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find all provision watchers associated to the DeviceProfile with the specified profile name.
   * List may be empty if no provision watchers match. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no DeviceProfile match
   * on the name provided.
   * 
   * @param profile name - provision watcher profile's name
   * @return List of ProvisionWatcher associated to the device profile
   */
  @RequestMapping(method = RequestMethod.GET, value = "/profilename/{profilename:.+}")
  @Override
  public List<ProvisionWatcher> watchersForProfileByName(@PathVariable String profilename) {
    try {
      DeviceProfile profile = profileDao.getByName(profilename);
      if (profile == null) {
        logger.error("Request for provision watcher by non-existent profile:  " + profilename);
        throw new NotFoundException(DeviceProfile.class.toString(), profilename);
      }
      return repos.findByProfile(profile);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find the provision watchers associated to the DeviceService with the specified service database
   * generated identifier. List may be empty if no provision watchers match. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if no DeviceService match on the id provided.
   * 
   * @param service id - device service's database generated identifier
   * @return ProvisionWatcher associated to the device service
   */
  @RequestMapping(method = RequestMethod.GET, value = "/service/{serviceId}")
  @Override
  public List<ProvisionWatcher> watcherForService(@PathVariable String serviceId) {
    try {
      DeviceService service = serviceDao.getById(serviceId);
      if (service == null) {
        logger.error("Request for provision watcher by non-existent service:  " + serviceId);
        throw new NotFoundException(DeviceService.class.toString(), serviceId);
      }
      return repos.findByService(service);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error("Error getting provision watchers by service:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find provision watchers associated to the DeviceService with the specified service name. List
   * may be empty if no provision watchers match. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) if no DeviceService match on the
   * name provided.
   * 
   * @param service name - provision watcher service's name
   * @return ProvisionWatcher associated to the device service
   */
  @RequestMapping(method = RequestMethod.GET, value = "/servicename/{servicename:.+}")
  @Override
  public List<ProvisionWatcher> watcherForServiceByName(@PathVariable String servicename) {
    try {
      DeviceService service = serviceDao.getByName(servicename);
      if (service == null) {
        logger.error("Request for provision watcher by non-existent service:  " + servicename);
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
   * Find ProvisionWatchers by an identifier key/value pair. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues.
   * 
   * @param key
   * @param value
   * @return
   */
  @RequestMapping(method = RequestMethod.GET, value = "/identifier/{key:.+}/{value:.+}")
  @Override
  public List<ProvisionWatcher> watchersForIdentifier(@PathVariable String key,
      @PathVariable String value) {
    try {
      return repos.findByIdendifierKeyValue("identifiers." + key, value);
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Add a new ProvisionWatcher - name must be unique. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues.Returns DataValidationException (HTTP 409) if an associated
   * object (Profile, Service) cannot be found with the id or name provided.
   * 
   * @param ProvisionWatcher to add
   * @return new database generated id for the new ProvisionWatcher
   */
  @RequestMapping(method = RequestMethod.POST)
  @Override
  public String add(@RequestBody ProvisionWatcher watcher) {
    if (watcher == null)
      throw new ServiceException(new DataValidationException("No watcher data provided"));
    try {
      attachAssociated(watcher);
      repos.save(watcher);
      notifyAssociates(watcher, Action.POST);
      return watcher.getId();
    } catch (DuplicateKeyException dE) {
      throw new DataValidationException("Name is not unique: " + watcher.getName());
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error adding provision watcher:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private void attachAssociated(ProvisionWatcher watcher) {
    DeviceService service = serviceDao.getByIdOrName(watcher.getService());
    if (service == null)
      throw new DataValidationException(
          "A provision watcher must be associated to a known device service.");
    watcher.setService(service);
    DeviceProfile profile = profileDao.getByIdOrName(watcher.getProfile());
    if (profile == null)
      throw new DataValidationException("A device must be associated to a known device profile.");
    watcher.setProfile(profile);
  }

  /**
   * Update the ProvisionWatcher identified by the id or name in the object provided. Id is used
   * first, name is used second for identification purposes. Watcher's service & profile cannot be
   * updated. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if no provision watcher with the provided id is found.
   * 
   * @param ProvisionWatcher object holding the identifier and new values for the ProvisionWatcher
   * @return boolean indicating success of the update
   */
  @RequestMapping(method = RequestMethod.PUT)
  @Override
  public boolean update(@RequestBody ProvisionWatcher watcher2) {
    if (watcher2 == null)
      throw new ServiceException(new DataValidationException("No provision watcher data provided"));
    try {
      ProvisionWatcher watcher = getByIdOrName(watcher2);
      if (watcher == null) {
        logger.error(
            "Request to update with non-existent or unidentified provision watcher (id/name):  "
                + watcher2.getId() + "/" + watcher2.getName());
        throw new NotFoundException(ProvisionWatcher.class.toString(), watcher2.getId());
      }
      updateWatcher(watcher2, watcher);
      notifyAssociates(watcher, Action.PUT);
      return true;
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error updating provision watcher:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private void updateWatcher(ProvisionWatcher from, ProvisionWatcher to) {
    if (from.getIdentifiers() != null)
      to.setIdentifiers(from.getIdentifiers());
    if (from.getOrigin() != 0)
      to.setOrigin(from.getOrigin());
    repos.save(to);
  }

  /**
   * Remove the ProvisionWatcher designated by the database generated id for the ProvisoinWatcher.
   * Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if no provision watcher with the provided id is found.
   * 
   * @param database generated id
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/id/{id}", method = RequestMethod.DELETE)
  @Override
  public boolean delete(@PathVariable String id) {
    try {
      ProvisionWatcher watcher = repos.findOne(id);
      if (watcher == null) {
        logger.error("Request to delete with non-existent provision watcher id:  " + id);
        throw new NotFoundException(ProvisionWatcher.class.toString(), id);
      }
      return delete(watcher);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error("Error removing provision watcher:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private boolean delete(ProvisionWatcher watcher) {
    repos.delete(watcher);
    notifyAssociates(watcher, Action.DELETE);
    return true;
  }

  /**
   * Remove the ProvisionWatcher designated by unique name identifier. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no
   * provision watcher with the provided name is found.
   * 
   * @param unique name of the ProvisionWatcher
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.DELETE)
  @Override
  public boolean deleteByName(@PathVariable String name) {
    try {
      ProvisionWatcher watcher = repos.findByName(name);
      if (watcher == null) {
        logger.error("Request to delete with unknown provision watcher name:  " + name);
        throw new NotFoundException(ProvisionWatcher.class.toString(), name);
      }
      return delete(watcher);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error("Error removing provision watcher:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private void notifyAssociates(ProvisionWatcher watcher, Action action) {
    callback.callback(watcher.getService(), watcher.getId(), action, ActionType.PROVISIONWATCHER);
  }

  private ProvisionWatcher getByIdOrName(ProvisionWatcher watcher) {
    if (watcher.getId() != null)
      return repos.findOne(watcher.getId());
    return repos.findByName(watcher.getName());
  }

}
