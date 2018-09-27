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
import java.util.Set;
import java.util.stream.Collectors;

import org.edgexfoundry.controller.DeviceServiceController;
import org.edgexfoundry.dao.AddressableDao;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.DeviceServiceDao;
import org.edgexfoundry.dao.DeviceServiceRepository;
import org.edgexfoundry.dao.ProvisionWatcherRepository;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.AdminState;
import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.domain.meta.OperatingState;
import org.edgexfoundry.domain.meta.ProvisionWatcher;
import org.edgexfoundry.exception.controller.ClientException;
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
@RequestMapping("/api/v1/deviceservice")
public class DeviceServiceControllerImpl implements DeviceServiceController {

  private static final org.edgexfoundry.support.logging.client.EdgeXLogger logger =
      org.edgexfoundry.support.logging.client.EdgeXLoggerFactory
          .getEdgeXLogger(DeviceServiceControllerImpl.class);

  private static final String ERR_GET = "Error getting service:  ";

  private static final String ERR_GET_SRV = "Error getting device service:  ";

  @Autowired
  private DeviceServiceRepository repos;

  @Autowired
  private DeviceServiceDao dao;

  @Autowired
  private AddressableDao addressableDao;

  @Autowired
  private DeviceRepository deviceRepos;

  @Autowired
  private ProvisionWatcherRepository watcherRepos;

  @Value("${read.max.limit}")
  private int maxLimit;

  /**
   * Fetch a specific device service by database generated id. May return null if no service with
   * the id is found. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * Returns NotFoundException (HTTP 404) if the device service is not found by the id provided.
   * 
   * @param String database generated id for the service
   * 
   * @return device service matching on id
   */
  @RequestMapping(method = RequestMethod.GET, value = "/{id}")
  @Override
  public DeviceService deviceService(@PathVariable String id) {
    try {
      DeviceService service = repos.findOne(id);
      if (service == null)
        throw new NotFoundException(DeviceService.class.toString(), id);
      return service;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return the DeviceService matching given name (service names should be unique). May be null if
   * no services matches on the name provided. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) if the device service is not found
   * by the name provided.
   * 
   * @param name
   * @return device service matching on name
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.GET)
  @Override
  public DeviceService deviceServiceForName(@PathVariable String name) {
    try {
      DeviceService service = repos.findByName(name);
      if (service == null)
        throw new NotFoundException(DeviceService.class.toString(), name);
      return service;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error("Error getting device profiles:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return all device services sorted by id. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns LimitExceededException (HTTP 413) if the number returned exceeds
   * the max limit.
   * 
   * @return list of profiles
   */
  @RequestMapping(method = RequestMethod.GET)
  @Override
  public List<DeviceService> deviceServices() {
    try {
      if (repos.count() > maxLimit) {
        logger.error("Max limit exceeded requesting device services");
        throw new LimitExceededException("DeviceProfile");
      }
      Sort sort = new Sort(Sort.Direction.DESC, "_id");
      return repos.findAll(sort);
    } catch (LimitExceededException lE) {
      throw lE;
    } catch (Exception e) {
      logger.error("Error getting device services:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find all device servicess associated to the Addressable with the specified addressable database
   * generated identifier. List may be empty if no device service match. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no
   * Addressable match on the id provided.
   * 
   * @param addressable id - addressable's database generated identifier
   * @return List of DeviceServices associated to the addressable
   */
  @RequestMapping(method = RequestMethod.GET, value = "/addressable/{addressableId}")
  @Override
  public List<DeviceService> deviceServicesForAddressable(@PathVariable String addressableId) {
    try {
      Addressable addressable = addressableDao.getById(addressableId);
      if (addressable == null) {
        logger.error("Request for device services by non-existent addressable:  " + addressableId);
        throw new NotFoundException(Addressable.class.toString(), addressableId);
      }
      return repos.findByAddressable(addressable);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET_SRV + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find all device serices associated to the Addressable with the specified addressable name. List
   * may be empty if no device services match. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) if no Addressable match on the name
   * provided.
   * 
   * @param addressable name - addressable's name
   * @return List of DeviceServices associated to the addressable
   */
  @RequestMapping(method = RequestMethod.GET, value = "/addressablename/{addressablename:.+}")
  @Override
  public List<DeviceService> deviceServicesForAddressableByName(
      @PathVariable String addressablename) {
    try {
      Addressable addressable = addressableDao.getByName(addressablename);
      if (addressable == null) {
        logger
            .error("Request for device services by non-existent addressable:  " + addressablename);
        throw new NotFoundException(Addressable.class.toString(), addressablename);
      }
      return repos.findByAddressable(addressable);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET_SRV + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find all DeviceServices having at least one label matching the label provided. List may be
   * empty if no device services match. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues.
   * 
   * @param label - label to be matched
   * 
   * @return List of DeviceService matching on specified label
   */
  @RequestMapping(method = RequestMethod.GET, value = "/label/{label:.+}")
  @Override
  public List<DeviceService> deviceServicesByLabel(@PathVariable String label) {
    try {
      return repos.findByLabelsIn(label);
    } catch (Exception e) {
      logger.error("Error getting services:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return a set (set versus list to insure element uniqueness) of addressables that are associated
   * to the devices of a device service. The device service is identified by id. Returns
   * NotFoundException (HTTP 404) if the device service is not found by id. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param id - database provided id for the device service
   * @return - set of unique addressable objects associated to devices of the owning device service
   */
  @RequestMapping(method = RequestMethod.GET, value = "/deviceaddressables/{id}")
  @Override
  public Set<Addressable> addressablesForAssociatedDevices(@PathVariable String id) {
    try {
      DeviceService service = repos.findOne(id);
      if (service == null)
        throw new NotFoundException(DeviceService.class.toString(), id);
      return deviceRepos.findByService(service).stream().map(d -> d.getAddressable())
          .collect(Collectors.toSet());
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return a set (set versus list to insure element uniqueness) of addressables that are associated
   * to the devices of a device service. The device service is identified by name. Returns
   * NotFoundException (HTTP 404) if the device service is not found by id. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param name - unique name for the device service
   * @return - set of unique addressable objects associated to devices of the owning device service
   */
  @RequestMapping(method = RequestMethod.GET, value = "/deviceaddressablesbyname/{name:.+}")
  @Override
  public Set<Addressable> addressablesForAssociatedDevicesByName(@PathVariable String name) {
    try {
      DeviceService service = repos.findByName(name);
      if (service == null)
        throw new NotFoundException(DeviceService.class.toString(), name);
      return deviceRepos.findByService(service).stream().map(d -> d.getAddressable())
          .collect(Collectors.toSet());
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Add a new DeviceService - name must be unique. Returns ServiceException (HTTP 503) for unknown
   * or unanticipated issues. Returns DataValidationException (HTTP 409) if an associated
   * addressable (by id or name) is not found.
   * 
   * @param DeviceService object
   * @return database generated identifier for the new device service
   */
  @RequestMapping(method = RequestMethod.POST)
  @Override
  public String add(@RequestBody DeviceService deviceService) {
    if (deviceService == null)
      throw new ServiceException(new DataValidationException("No device service data provided"));
    try {
      attachAssociated(deviceService);
      repos.save(deviceService);
      return deviceService.getId();
    } catch (DuplicateKeyException dE) {
      throw new DataValidationException("Name is not unique: " + deviceService.getName());
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error adding device service:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private void attachAssociated(DeviceService service) {
    try {
      Addressable addressable = addressableDao.getByIdOrName(service.getAddressable());
      if (addressable == null)
        throw new DataValidationException(
            "A device service must be associated to a known addressable.");
      service.setAddressable(addressable);
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error getting addressables:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the last connected time of the device service by database generated identifier. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device service cannot be found by the identifier provided.
   * 
   * @param id - database generated identifier for the device service
   * @param time - new last connected time in milliseconds
   * @return boolean indicating success of update
   */
  @RequestMapping(value = "/{id}/lastconnected/{time}", method = RequestMethod.PUT)
  public boolean updateLastConnected(@PathVariable String id, @PathVariable long time) {
    try {
      DeviceService deviceService = repos.findOne(id);
      if (deviceService == null) {
        logger.error(
            "Request to update last connected time with non-existent device service:  " + id);
        throw new NotFoundException(DeviceService.class.toString(), id);
      }
      return updateLastConnected(deviceService, time);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET_SRV + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private boolean updateLastConnected(DeviceService deviceService, long time) {
    try {
      deviceService.setLastConnected(time);
      repos.save(deviceService);
      return true;
    } catch (Exception e) {
      logger.error("Error updating last connected time for the device service:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the last connected time of the device service by unique name of the device service.
   * Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if the device service cannot be found by the name provided.
   * 
   * @param name - device service name
   * @param time - new last connected time in milliseconds
   * @return boolean indicating success of update
   */
  @RequestMapping(value = "/name/{name:.+}/lastconnected/{time}", method = RequestMethod.PUT)
  @Override
  public boolean updateLastConnectedByName(@PathVariable String name, @PathVariable long time) {
    try {
      DeviceService deviceService = repos.findByName(name);
      if (deviceService == null) {
        logger.error(
            "Request to update last connected time with non-existent device service:  " + name);
        throw new NotFoundException(DeviceService.class.toString(), name);
      }
      return updateLastConnected(deviceService, time);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET_SRV + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the last reported time of the device service by database generated identifier. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device service cannot be found by the identifier provided.
   * 
   * @param id - database generated identifier for the device service
   * @param time - new last reported time in milliseconds
   * @return boolean indicating success of update
   */
  @RequestMapping(value = "/{id}/lastreported/{time}", method = RequestMethod.PUT)
  @Override
  public boolean updateLastReported(@PathVariable String id, @PathVariable long time) {
    try {
      DeviceService deviceService = repos.findOne(id);
      if (deviceService == null) {
        logger
            .error("Request to update last reported time with non-existent device service:  " + id);
        throw new NotFoundException(DeviceService.class.toString(), id);
      }
      return updateLastReported(deviceService, time);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET_SRV + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private boolean updateLastReported(DeviceService deviceService, long time) {
    try {
      deviceService.setLastReported(time);
      repos.save(deviceService);
      return true;
    } catch (Exception e) {
      logger.error("Error updating last reported time for the device service:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the last reported time of the device service by unique name of the device service.
   * Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if the device service cannot be found by the name provided.
   * 
   * @param name - device service name
   * @param time - new last reported time in milliseconds
   * @return boolean indicating success of update
   */
  @RequestMapping(value = "/name/{name:.+}/lastreported/{time}", method = RequestMethod.PUT)
  @Override
  public boolean updateLastReportedByName(@PathVariable String name, @PathVariable long time) {
    try {
      DeviceService deviceService = repos.findByName(name);
      if (deviceService == null) {
        logger.error(
            "Request to update last reported time with non-existent device service:  " + name);
        throw new NotFoundException(DeviceService.class.toString(), name);
      }
      return updateLastReported(deviceService, time);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET_SRV + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the op state of the device service by database generated identifier. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device service cannot be found by the identifier provided.
   * 
   * @param id - database generated identifier for the device service
   * @param opState - new op state for the device service (either ENABLED or DISABLED)
   * @return - boolean indicating success of the operation
   */
  @RequestMapping(value = "/{id}/opstate/{opState}", method = RequestMethod.PUT)
  @Override
  public boolean updateOpState(@PathVariable String id, @PathVariable String opState) {
    try {
      DeviceService deviceService = repos.findOne(id);
      if (deviceService == null) {
        logger.error("Request to update op state with non-existent device service:  " + id);
        throw new NotFoundException(DeviceService.class.toString(), id);
      }
      return updateOpState(deviceService, opState);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET_SRV + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private boolean updateOpState(DeviceService deviceService, String state) {
    try {
      deviceService.setOperatingState(OperatingState.valueOf(state));
      repos.save(deviceService);
      return true;
    } catch (Exception e) {
      logger.error("Error updating op state for the device service:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the op status time of the device service by unique name of the device service. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device service cannot be found by the name provided.
   * 
   * @param name - device service name
   * @param opState - new op state for the device service (either ENABLED or DISABLED)
   * @return - boolean indicating success of the operation
   */
  @RequestMapping(value = "/name/{name:.+}/opstate/{opState}", method = RequestMethod.PUT)
  @Override
  public boolean updateOpStateByName(@PathVariable String name, @PathVariable String opState) {
    try {
      DeviceService deviceService = repos.findByName(name);
      if (deviceService == null) {
        logger.error("Request to update op state with non-existent device service name:  " + name);
        throw new NotFoundException(DeviceService.class.toString(), name);
      }
      return updateOpState(deviceService, opState);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET_SRV + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the admin state of the device service by database generated identifier. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device service cannot be found by the identifier provided.
   * 
   * @param id - database generated identifier for the device service
   * @param adminstate - new admin state for the device service (either LOCKED or UNLOCKED)
   * @return - boolean indicating success of the operation
   */
  @RequestMapping(value = "/{id}/adminstate/{adminState}", method = RequestMethod.PUT)
  @Override
  public boolean updateAdminState(@PathVariable String id, @PathVariable String adminState) {
    try {
      DeviceService deviceService = repos.findOne(id);
      if (deviceService == null) {
        logger.error("Request to update admin state with non-existent device service:  " + id);
        throw new NotFoundException(DeviceService.class.toString(), id);
      }
      return updateAdminState(deviceService, adminState);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET_SRV + e.getMessage());
      throw new ServiceException(e);
    }
  }


  private boolean updateAdminState(DeviceService deviceService, String state) {
    try {
      deviceService.setAdminState(AdminState.valueOf(state));
      repos.save(deviceService);
      return true;
    } catch (Exception e) {
      logger.error("Error updating admin state for the device service:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the admin state of the device service by device service name. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if the
   * device service cannot be found by the identifier provided.
   * 
   * @param name - device service name
   * @param opState - new admin state for the device service (either LOCKED or UNLOCKED)
   * @return - boolean indicating success of the operation
   */
  @RequestMapping(value = "/name/{name:.+}/adminstate/{adminState}", method = RequestMethod.PUT)
  @Override
  public boolean updateAdminStateByName(@PathVariable String name,
      @PathVariable String adminState) {
    try {
      DeviceService deviceService = repos.findByName(name);
      if (deviceService == null) {
        logger
            .error("Request to update admin state with non-existent device service name:  " + name);
        throw new NotFoundException(DeviceService.class.toString(), name);
      }
      return updateAdminState(deviceService, adminState);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET_SRV + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the DeviceServcie identified by the id or name stored in the object provided. Id is used
   * first, name is used second for identification purposes. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if the device service
   * cannot be found by the identifier provided.
   * 
   * @param deviceService2 - object holding the identifier and new values for the DeviceService
   * @return boolean indicating success of the update
   */
  @RequestMapping(method = RequestMethod.PUT)
  @Override
  public boolean update(@RequestBody DeviceService deviceService2) {
    if (deviceService2 == null)
      throw new ServiceException(new DataValidationException("No device service data provided"));
    try {
      DeviceService deviceService = dao.getByIdOrName(deviceService2);
      if (deviceService == null) {
        logger
            .error("Request to update with non-existent or unidentified device service (id/name):  "
                + deviceService2.getId() + "/" + deviceService2.getName());
        throw new NotFoundException(DeviceService.class.toString(), deviceService2.getId());
      }
      updateDeviceService(deviceService2, deviceService);
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

  private void updateDeviceService(DeviceService from, DeviceService to) {
    checkAddressable(from, to);
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
    if (from.getName() != null)
      to.setName(from.getName());
    if (from.getOperatingState() != null)
      to.setOperatingState(from.getOperatingState());
    if (from.getOrigin() != 0)
      to.setOrigin(from.getOrigin());
    repos.save(to);
  }
  
  private void checkAddressable(DeviceService from, DeviceService to) {
    if (from.getAddressable() != null) {
      Addressable addr = addressableDao.getByIdOrName(from.getAddressable());
      if (addr != null)
        to.setAddressable(addr);
      else {
        logger.error(
            "Unable to locate addressable and cannot set device service addressable to null");
        if (from.getAddressable().getId() != null)
          throw new NotFoundException(Addressable.class.toString(), from.getAddressable().getId());
        else
          throw new NotFoundException(Addressable.class.toString(),
              from.getAddressable().getName());
      }
    }
  }

  /**
   * Remove the DeviceService designated by database generated id. Returns ServiceException (HTTP
   * 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if the device
   * service cannot be found by the identifier provided.
   * 
   * 
   * @param database generated id for the device service
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/id/{id}", method = RequestMethod.DELETE)
  @Override
  public boolean delete(@PathVariable String id) {
    try {
      DeviceService deviceService = repos.findOne(id);
      if (deviceService == null) {
        logger.error("Request to delete with non-existent device service by id:  " + id);
        throw new NotFoundException(DeviceService.class.toString(), id);
      }
      return deleteDeviceService(deviceService);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error removing device service:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Remove the DeviceService designated by name. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) if the device service cannot be
   * found by the name provided.
   * 
   * 
   * @param name for the device service
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.DELETE)
  @Override
  public boolean deleteByName(@PathVariable String name) {
    try {
      DeviceService deviceService = repos.findByName(name);
      if (deviceService == null) {
        logger.error("Request to delete with unknown device service by name:  " + name);
        throw new NotFoundException(DeviceService.class.toString(), name);
      }
      return deleteDeviceService(deviceService);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error removing device service:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private boolean deleteDeviceService(DeviceService deviceService) {
    deleteAssociatedDevices(deviceService);
    deleteAssociatedProvisionWatchers(deviceService);
    repos.delete(deviceService);
    return true;
  }

  private void deleteAssociatedDevices(DeviceService service) {
    List<Device> devices = deviceRepos.findByService(service);
    if (devices != null)
      devices.stream().forEach(d -> deviceRepos.delete(d));
  }

  private void deleteAssociatedProvisionWatchers(DeviceService service) {
    List<ProvisionWatcher> watchers = watcherRepos.findByService(service);
    if (watchers != null)
      watchers.stream().forEach(w -> watcherRepos.delete(w));
  }

}
